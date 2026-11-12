//! The boundary of a program, slot by slot.
//!
//! torch's contract with a backend is one FX graph: the compiled callable
//! takes one value per placeholder, in order, and returns one value per
//! leaf of the graph's output, in order. A slot is one of those positions.
//! Its kind is the type of the value torch passes or expects there: a
//! tensor, an `int`, a `float`, a `bool`, or any other object.
//!
//! The plan is read off the serialized program alone, before translation
//! and without it, so a frontend can ask for it first and state only what
//! the plan says the program touches.
//!
//! Arguments follow torch's own rule for placeholders. A tensor is storage
//! the program binds. A symbolic scalar is the expression torch recorded
//! for it: every node that reads it reads that expression, and the call's
//! value is read in one case only, an integer argument that IS a symbol,
//! which defines that symbol for the call. An argument no node reads keeps
//! its slot and is never touched.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};
use luminal::prelude::{DynMap, IntExpr};

use crate::pt2_parser::{InputKind, ParsedPT2};
use crate::pt2_schema::{Argument, InputSpec, OutputSpec};

/// What the value in a slot is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    Tensor,
    Int,
    Float,
    Bool,
    /// `None`, a string, or an opaque object.
    Object,
}

impl SlotKind {
    pub fn name(self) -> &'static str {
        match self {
            SlotKind::Tensor => "tensor",
            SlotKind::Int => "int",
            SlotKind::Float => "float",
            SlotKind::Bool => "bool",
            SlotKind::Object => "object",
        }
    }
}

/// One argument of the program: one placeholder, in call order. A tensor
/// constant the program holds is not an argument and has no slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSlot {
    /// The placeholder's name; empty for a constant argument, which has
    /// none.
    pub graph_name: String,
    pub kind: SlotKind,
    /// Whether a node takes the argument as an operand, or the program
    /// writes back into it. A tensor that is read is storage the program
    /// binds.
    pub read: bool,
    /// For an integer argument that IS a symbol the program uses: the PT2
    /// name of that symbol, which the call's value defines. A compound
    /// argument (`2*s0`) is a statement about symbols other arguments
    /// define, and its call-time value is not read.
    pub binds: Option<String>,
}

/// Where the value of an output leaf comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSource {
    /// The program computes it: its row in [`BoundaryPlan::rows`].
    Program { row: usize },
    /// The leaf is an argument, so the caller's own object is returned:
    /// its slot in [`BoundaryPlan::inputs`].
    Input { slot: usize },
    /// A symbolic scalar computed from sizes, evaluated per call from the
    /// expression torch recorded for it.
    Expression,
    /// The value is written in the program: `None`, or a constant.
    Literal,
}

/// One leaf of the program's output, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSlot {
    /// The leaf's value name; empty for a literal.
    pub graph_name: String,
    pub kind: SlotKind,
    pub source: OutputSource,
}

/// One tensor the program writes: a returned value, or a writeback into
/// an input's storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputRow {
    pub graph_name: String,
    /// The graph output position that states this value.
    pub position: usize,
    /// For a mutation: the graph name of the input it writes into.
    pub mutation_target: Option<String>,
    /// Whether an output leaf returns this value. A mutation-only sink is
    /// `false`; a mutated input the program also returns is one row with
    /// `returned` set and a target.
    pub returned: bool,
}

/// A condition torch asserts at run time: a symbolic bool that must hold
/// at every call, with the message torch raises when it does not.
///
/// torch states one as a comparison of symbolic integers and an
/// `_assert_scalar` node over it. A declared range arrives this way too,
/// and Dynamo guards those before the program is called, but a check the
/// user wrote (`torch._check(n != 5)`) is asserted nowhere else.
#[derive(Debug, Clone)]
pub struct RuntimeCheck {
    /// The condition read as 1 or 0 over the program's symbols.
    pub condition: IntExpr,
    pub message: String,
}

/// A program's boundary: its arguments, its output leaves, the tensors it
/// writes, and the conditions a call must meet.
#[derive(Debug, Clone, Default)]
pub struct BoundaryPlan {
    pub inputs: Vec<InputSlot>,
    pub outputs: Vec<OutputSlot>,
    pub rows: Vec<OutputRow>,
    /// The runtime checks the translator could read. They are separate
    /// from binding: a call binds the same way whether or not they run.
    pub checks: Vec<RuntimeCheck>,
}

impl BoundaryPlan {
    /// The arguments as plain rows: `(kind, graph name, read, symbol)`.
    pub fn input_rows(&self) -> Vec<(String, String, bool, Option<String>)> {
        self.inputs
            .iter()
            .map(|slot| {
                (
                    slot.kind.name().to_string(),
                    slot.graph_name.clone(),
                    slot.read,
                    slot.binds.clone(),
                )
            })
            .collect()
    }

    /// The output leaves as plain rows: `(kind, source, index)`. `source`
    /// is `"program"` (index: the row), `"input"` (index: the argument
    /// slot), `"expression"` or `"literal"`.
    pub fn output_rows(&self) -> Vec<(String, String, Option<usize>)> {
        self.outputs
            .iter()
            .map(|slot| {
                let (source, index) = match slot.source {
                    OutputSource::Program { row } => ("program", Some(row)),
                    OutputSource::Input { slot } => ("input", Some(slot)),
                    OutputSource::Expression => ("expression", None),
                    OutputSource::Literal => ("literal", None),
                };
                (slot.kind.name().to_string(), source.to_string(), index)
            })
            .collect()
    }

    /// The written tensors as plain rows: `(graph name, mutation target,
    /// returned)`.
    pub fn written_rows(&self) -> Vec<(String, Option<String>, bool)> {
        self.rows
            .iter()
            .map(|row| {
                (
                    row.graph_name.clone(),
                    row.mutation_target.clone(),
                    row.returned,
                )
            })
            .collect()
    }

    /// The tensor arguments no node reads, by graph name. Nothing states
    /// or checks their device, layout or storage.
    pub fn unread_tensors(&self) -> HashSet<&str> {
        self.inputs
            .iter()
            .filter(|slot| slot.kind == SlotKind::Tensor && !slot.read)
            .map(|slot| slot.graph_name.as_str())
            .collect()
    }

    /// The first runtime check a call's symbol values break. A check
    /// over a symbol the call did not bind is not evaluated.
    pub fn failed_check(&self, dims: &DynMap) -> Option<&RuntimeCheck> {
        self.checks
            .iter()
            .find(|check| check.condition.exec(dims) == Some(0))
    }

    /// The argument slot a name occupies.
    pub fn argument(&self, graph_name: &str) -> Option<(usize, &InputSlot)> {
        self.inputs
            .iter()
            .enumerate()
            .find(|(_, slot)| !slot.graph_name.is_empty() && slot.graph_name == graph_name)
    }
}

/// The kind of a constant in a signature argument, when it is one.
fn constant_kind(arg: &Argument) -> Option<SlotKind> {
    match arg {
        Argument::Int(_) => Some(SlotKind::Int),
        Argument::Float(_) => Some(SlotKind::Float),
        Argument::Bool(_) => Some(SlotKind::Bool),
        Argument::Other(value) => constant_kind_of_json(value),
        _ => None,
    }
}

fn constant_kind_of_json(value: &serde_json::Value) -> Option<SlotKind> {
    let object = value.as_object()?;
    [
        ("as_none", SlotKind::Object),
        ("as_int", SlotKind::Int),
        ("as_float", SlotKind::Float),
        ("as_bool", SlotKind::Bool),
        ("as_string", SlotKind::Object),
    ]
    .into_iter()
    .find(|(key, _)| object.contains_key(*key))
    .map(|(_, kind)| kind)
}

/// Read a program's boundary off its serialized form.
pub fn plan(parsed: &ParsedPT2) -> Result<BoundaryPlan> {
    let graph = &parsed.program.graph_module.graph;
    let signature = &parsed.program.graph_module.signature;

    // Every value name a node takes as an operand, and the symbol each
    // integer argument defines.
    let operands = parsed.operand_names();
    let definitions = parsed.integer_symbol_definitions();

    // Held inputs (lifted tensor constants) are not arguments.
    let held: HashSet<String> = parsed
        .classify_inputs()
        .into_iter()
        .filter_map(|kind| match kind {
            InputKind::Buffer { graph_name, .. } => Some(graph_name),
            InputKind::UserInput { .. } => None,
        })
        .collect();

    // The inputs a mutation writes into are read: the program binds their
    // storage.
    let mut written: HashSet<&str> = HashSet::new();
    for spec in &signature.output_specs {
        if let OutputSpec::UserInputMutation(mutation) = spec {
            written.insert(mutation.user_input_name.as_str());
        }
    }

    let mut inputs = Vec::new();
    for (position, tref) in graph.inputs.iter().enumerate() {
        let name = tref.value_name().unwrap_or_default();
        if !name.is_empty() && held.contains(name) {
            continue;
        }
        let (kind, binds) = if tref.as_tensor.is_some() {
            (SlotKind::Tensor, None)
        } else if let Some(scalar) = &tref.as_sym_int {
            (SlotKind::Int, definitions.get(&scalar.as_name).cloned())
        } else if tref.as_sym_float.is_some() {
            (SlotKind::Float, None)
        } else if tref.as_sym_bool.is_some() {
            (SlotKind::Bool, None)
        } else {
            // A constant argument: the signature states what it is.
            let kind = match signature.input_specs.get(position) {
                Some(InputSpec::UserInput(spec)) => constant_kind(&spec.arg),
                _ => None,
            };
            (kind.unwrap_or(SlotKind::Object), None)
        };
        inputs.push(InputSlot {
            graph_name: name.to_string(),
            kind,
            read: !name.is_empty() && (operands.contains(name) || written.contains(name)),
            binds,
        });
    }

    let mut plan = BoundaryPlan {
        inputs,
        ..Default::default()
    };

    // Export node name of each mutation row -> its row.
    let mut writeback_by_name: HashMap<&str, usize> = HashMap::new();
    for (position, tref) in graph.outputs.iter().enumerate() {
        let spec = signature.output_specs.get(position);
        let mutation_target = match spec {
            Some(OutputSpec::UserInputMutation(mutation)) => Some(mutation.user_input_name.clone()),
            _ => None,
        };

        // A mutation writes an input's storage and is not an output leaf.
        // torch emits it before the user output that returns the same
        // node, and that user output IS the writeback.
        if mutation_target.is_some() {
            let name = tref
                .value_name()
                .ok_or_else(|| anyhow!("graph output has no tensor name: {tref:?}"))?;
            writeback_by_name.insert(name, plan.rows.len());
            plan.rows.push(OutputRow {
                graph_name: name.to_string(),
                position,
                mutation_target,
                returned: false,
            });
            continue;
        }

        // A literal leaf has no value name: the signature states it.
        if let Some(OutputSpec::UserOutput(output)) = spec
            && let Some(kind) = constant_kind_of_json(&output.arg)
        {
            plan.outputs.push(OutputSlot {
                graph_name: String::new(),
                kind,
                source: OutputSource::Literal,
            });
            continue;
        }

        let name = tref
            .value_name()
            .ok_or_else(|| anyhow!("graph output has no tensor name: {tref:?}"))?;
        let symbolic = if tref.as_sym_int.is_some() {
            Some(SlotKind::Int)
        } else if tref.as_sym_float.is_some() {
            Some(SlotKind::Float)
        } else if tref.as_sym_bool.is_some() {
            Some(SlotKind::Bool)
        } else {
            None
        };
        let (kind, source) = if let Some(&row) = writeback_by_name.get(name) {
            plan.rows[row].returned = true;
            (SlotKind::Tensor, OutputSource::Program { row })
        } else if let Some((slot, argument)) = plan.argument(name) {
            (argument.kind, OutputSource::Input { slot })
        } else if let Some(kind) = symbolic {
            (kind, OutputSource::Expression)
        } else {
            // Two leaves naming different nodes are two rows even when they
            // translate to one value: each gets its own buffer.
            let row = plan.rows.len();
            plan.rows.push(OutputRow {
                graph_name: name.to_string(),
                position,
                mutation_target: None,
                returned: true,
            });
            (SlotKind::Tensor, OutputSource::Program { row })
        };
        plan.outputs.push(OutputSlot {
            graph_name: name.to_string(),
            kind,
            source,
        });
    }
    Ok(plan)
}
