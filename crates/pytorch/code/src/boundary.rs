//! The boundary tables of a code object: torch's side of a call.
//!
//! These say what a call's arguments are and where its outputs come from.
//! They say nothing about how a backend reads a tensor or what it runs, so
//! they are the same for every backend.

use anyhow::{Result, bail, ensure};
use luminal::prelude::{DynMap, IntExpr, Symbol};
use luminal_torch_slots::SlotKind;
use serde::{Deserialize, Serialize};

/// The part of a code object that every backend states the same way. A
/// backend's code object holds one flattened beside its own tables, so the
/// saved form has these fields at its top level.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Boundary {
    /// What kind of value each argument slot holds, in call order.
    pub slots: Vec<SlotKind>,
    /// Each slot's graph name, which a refusal names the argument by. Empty
    /// for a constant argument, which has none.
    pub slot_names: Vec<String>,
    /// What a call does with each slot. One per slot.
    pub actions: Vec<SlotAction>,
    /// One per output leaf, in torch's order.
    pub leaves: Vec<Leaf>,
    /// The conditions torch asserted on the call's dimensions.
    pub checks: Vec<Check>,
    /// The value each dimension has until a call states it.
    pub dims: Vec<(Symbol, usize)>,
}

/// What a call does with one argument slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotAction {
    /// A tensor the program reads, as input `input` of the backend's own
    /// input table. Its shape states dimensions. How it is read (bound in
    /// place, or copied) is the backend's.
    Read { input: usize },
    /// An integer that IS a dimension: the call's value defines it.
    Dimension { symbol: Symbol },
    /// Not looked at: a tensor no node reads, or a scalar the program reads
    /// as the expression torch recorded for it.
    Skip,
}

/// Where the value of one output leaf comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Leaf {
    /// The tensor in a row of the backend's own row table. `scalar` unboxes
    /// a rank-zero tensor to the Python scalar the graph returned.
    Row {
        row: usize,
        scalar: bool,
    },
    /// The caller's own argument, returned as it is.
    Argument {
        slot: usize,
    },
    /// An integer computed from the call's arguments.
    Integer {
        expr: IntExpr,
        sources: Vec<Source>,
    },
    /// A bool computed from the call's arguments: a condition read as 1 or
    /// 0 over the dimensions.
    Bool {
        expr: IntExpr,
        sources: Vec<Source>,
    },
    Literal(Literal),
}

/// Where a call states one dimension an output expression names: an integer
/// argument that is the dimension (`axis` is `None`), or one axis of a
/// tensor argument.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub symbol: Symbol,
    pub slot: usize,
    pub axis: Option<usize>,
}

/// A value written in the program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Literal {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

/// A condition torch asserts at run time, read as 1 or 0 over the program's
/// dimensions, with the message torch raises when it does not hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Check {
    pub condition: IntExpr,
    pub message: String,
}

impl Boundary {
    /// The boundary of a graph that reads no argument: every slot is
    /// skipped, and the leaves forward arguments, sizes and constants.
    pub fn forwarding(slots: Vec<SlotKind>, slot_names: Vec<String>, leaves: Vec<Leaf>) -> Self {
        Self {
            actions: vec![SlotAction::Skip; slots.len()],
            slots,
            slot_names,
            leaves,
            checks: Vec::new(),
            dims: Vec::new(),
        }
    }

    /// What must hold for the tables to index one another and the backend's
    /// own. Checked once at load, so a call never indexes out of range.
    ///
    /// `arguments` has one entry per input of the backend's input table:
    /// whether a call supplies it (`true`) or the program holds it as a
    /// constant. `rows` is the length of the backend's row table.
    pub fn validate(&self, arguments: &[bool], rows: usize) -> Result<()> {
        if self.actions.len() != self.slots.len() || self.slot_names.len() != self.slots.len() {
            bail!(
                "a code object states {} slots and {} slot actions",
                self.slots.len(),
                self.actions.len()
            );
        }
        for (slot, action) in self.actions.iter().enumerate() {
            if let SlotAction::Read { input } = action {
                ensure!(
                    arguments.get(*input) == Some(&true),
                    "slot {slot} reads input {input}, which is not an argument of the code \
                     object"
                );
            }
        }
        for (position, leaf) in self.leaves.iter().enumerate() {
            let in_range = match leaf {
                Leaf::Row { row, .. } => *row < rows,
                Leaf::Argument { slot } => *slot < self.slots.len(),
                Leaf::Integer { sources, .. } | Leaf::Bool { sources, .. } => {
                    sources.iter().all(|source| source.slot < self.slots.len())
                }
                Leaf::Literal(_) => true,
            };
            if !in_range {
                bail!("output leaf {position} names a row or slot the code object does not have");
            }
        }
        Ok(())
    }

    /// The value every dimension has before any call states it.
    pub fn state(&self) -> DynMap {
        self.dims.iter().copied().collect()
    }

    /// The first condition torch asserted that does not hold at `dims`. A
    /// condition over a dimension no call has stated is not evaluated.
    pub fn failed_check(&self, dims: &DynMap) -> Option<&Check> {
        self.checks
            .iter()
            .find(|check| check.condition.exec(dims) == Some(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Boundary {
        let n = Symbol::new("s0");
        Boundary {
            slots: vec![SlotKind::Tensor, SlotKind::Int, SlotKind::Tensor],
            slot_names: vec!["x".into(), "n".into(), "unused".into()],
            actions: vec![
                SlotAction::Read { input: 0 },
                SlotAction::Dimension { symbol: n },
                SlotAction::Skip,
            ],
            leaves: vec![
                Leaf::Row {
                    row: 0,
                    scalar: false,
                },
                Leaf::Argument { slot: 2 },
                Leaf::Integer {
                    expr: IntExpr::from(n) * IntExpr::from(2i64),
                    sources: vec![Source {
                        symbol: n,
                        slot: 0,
                        axis: Some(0),
                    }],
                },
                Leaf::Literal(Literal::None),
                Leaf::Literal(Literal::Float(1.5)),
            ],
            checks: vec![Check {
                condition: IntExpr::from(n),
                message: "n must not be zero".into(),
            }],
            dims: vec![(n, 3)],
        }
    }

    /// The saved form reads back as the same tables, and writing it again
    /// gives the same text: nothing in the tables depends on the process
    /// that wrote them.
    #[test]
    fn the_tables_round_trip_through_their_saved_form() {
        let boundary = sample();
        let text = serde_json::to_string(&boundary).unwrap();
        let read: Boundary = serde_json::from_str(&text).unwrap();
        assert_eq!(serde_json::to_string(&read).unwrap(), text);
        assert_eq!(read.actions, boundary.actions);
        assert_eq!(read.dims, boundary.dims);
        assert!(
            text.contains(r#""slots":["tensor","int","tensor"]"#),
            "{text}"
        );
        assert!(text.contains(r#"{"read":{"input":0}}"#), "{text}");
        match (&read.leaves[2], &boundary.leaves[2]) {
            (
                Leaf::Integer { expr, sources },
                Leaf::Integer {
                    expr: wanted,
                    sources: wanted_sources,
                },
            ) => {
                assert_eq!(expr, wanted);
                assert_eq!(sources, wanted_sources);
            }
            other => panic!("leaf 2 read back as {other:?}"),
        }
    }

    /// Tables that index outside one another, or outside the backend's own,
    /// are refused at load, so a call never has to check.
    #[test]
    fn tables_that_disagree_are_refused() {
        // Input 0 is an argument, input 1 a constant; one row.
        let arguments = [true, false];
        sample().validate(&arguments, 1).unwrap();

        let mut boundary = sample();
        boundary.leaves.push(Leaf::Row {
            row: 7,
            scalar: false,
        });
        let error = boundary.validate(&arguments, 1).unwrap_err();
        assert!(error.to_string().contains("output leaf 5"), "{error}");

        let mut boundary = sample();
        boundary.actions.pop();
        let error = boundary.validate(&arguments, 1).unwrap_err();
        assert!(error.to_string().contains("3 slots and 2"), "{error}");

        // A slot reads an argument, never a constant the program holds.
        let mut boundary = sample();
        boundary.actions[0] = SlotAction::Read { input: 1 };
        let error = boundary.validate(&arguments, 1).unwrap_err();
        assert!(
            error.to_string().contains("slot 0 reads input 1"),
            "{error}"
        );
        let mut boundary = sample();
        boundary.actions[0] = SlotAction::Read { input: 2 };
        assert!(boundary.validate(&arguments, 1).is_err());
    }

    #[test]
    fn a_check_fails_only_at_dimensions_that_break_it() {
        let boundary = sample();
        let n = boundary.dims[0].0;
        assert!(boundary.failed_check(&boundary.state()).is_none());
        let zero: DynMap = [(n, 0)].into_iter().collect();
        let failed = boundary.failed_check(&zero).expect("n is zero");
        assert_eq!(failed.message, "n must not be zero");
        // A dimension no call has stated is not evaluated.
        assert!(boundary.failed_check(&DynMap::default()).is_none());
    }
}
