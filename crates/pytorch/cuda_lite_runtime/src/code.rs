//! The CUDA backend's code object: one compiled graph as plain data.
//!
//! A code object is everything a call needs and nothing the compiler needed.
//! It holds no parsed program, no translated graph, no searched plan and no
//! op: the program is the list of kernel launches, copies and library calls
//! the device runs, with every kernel already compiled. The compiler builds
//! one at the end of search and hands it over as bytes ([`Code::to_bytes`]);
//! every executable, in process or loaded from a file, is built from those
//! bytes ([`Code::from_bytes`]).

use anyhow::{Context, Result, ensure};
use luminal::prelude::IntExpr;
use luminal_cuda_lite::lowered::LoweredProgram;
use luminal_torch_code::{Boundary, hex};
use luminal_torch_slots::TorchDtype;
use serde::{Deserialize, Serialize};

/// The version of the byte form. A program saved under another version is
/// refused by number, never read as this one.
pub const FORMAT: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct Code {
    pub format: u32,
    /// Torch's side of a call, which every backend states the same way:
    /// the argument slots, the output leaves, the conditions torch asserted
    /// and the value each dimension starts at. A slot that is read is
    /// bound: its device memory is the program's input, in place.
    #[serde(flatten)]
    pub boundary: Boundary,
    /// The tensors the program reads, each bound in place on its own
    /// buffer: the arguments it reads and the constants it holds.
    pub inputs: Vec<Input>,
    /// The tensors the program writes, in row order. Row `i` is output slot
    /// `i` of the program.
    pub rows: Vec<Row>,
    /// What the device runs. `None` for a graph that computes nothing: it
    /// forwards arguments, sizes and constants only.
    pub program: Option<LoweredProgram>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Input {
    /// The graph name, which every refusal leads with.
    pub name: String,
    /// The program buffer the tensor's memory is bound on.
    pub buffer: i64,
    /// One extent per axis: a literal, or an expression over dimensions.
    pub shape: Vec<IntExpr>,
    /// Whether a row writes back into this tensor.
    pub written: bool,
    /// For a constant the program holds: the tensor itself. An argument has
    /// none; a call supplies it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constant: Option<Constant>,
}

/// A tensor the program holds. Its memory is allocated on the device once,
/// when the executable first runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constant {
    pub dtype: TorchDtype,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    /// The bytes its strides reach from its first element, in storage
    /// order.
    #[serde(with = "hex")]
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub name: String,
    pub dtype: TorchDtype,
    pub shape: Vec<IntExpr>,
    /// The element strides the caller receives the row at: the ones eager
    /// gives it.
    pub strides: Strides,
    /// The program buffer the row is written on.
    pub buffer: i64,
    pub storage: Storage,
}

/// The element strides of a row, at a call's dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strides {
    RowMajor,
    ColumnMajor,
    /// One stride per axis, each an expression over the dimensions.
    Elements(Vec<IntExpr>),
}

/// Whose memory a row is written in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Storage {
    /// A tensor torch allocates for this call.
    Fresh,
    /// The caller's own argument: a mutation, written in place. That
    /// argument is the row's value.
    WritesBack { slot: usize },
    /// A view of a tensor the call already holds, `offset` elements past
    /// that tensor's first element. Nothing is allocated.
    View { of: Owner, offset: i64 },
}

/// The tensor a view row shares storage with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    Argument { slot: usize },
    Constant { input: usize },
    Row { row: usize },
}

impl Code {
    /// The byte form. A program that cannot be written out (a kernel that
    /// was never compiled, a library call with no record) is refused by
    /// name.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        if let Some(program) = &self.program {
            program.ensure_portable()?;
        }
        serde_json::to_vec(self).context("writing a CUDA code object")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let code: Code = luminal_torch_code::bytes::read(bytes, FORMAT, "CUDA")?;
        code.validate()?;
        Ok(code)
    }

    /// What must hold for the tables to index one another and the program.
    /// Checked once at load, so a call never indexes out of range.
    pub fn validate(&self) -> Result<()> {
        let arguments: Vec<bool> = self
            .inputs
            .iter()
            .map(|input| input.constant.is_none())
            .collect();
        self.boundary.validate(&arguments, self.rows.len())?;
        let slots = self.boundary.slots.len();
        for (index, row) in self.rows.iter().enumerate() {
            let in_range = match &row.storage {
                Storage::Fresh => true,
                Storage::WritesBack { slot } => *slot < slots,
                Storage::View { of, .. } => match of {
                    Owner::Argument { slot } => *slot < slots,
                    Owner::Constant { input } => self
                        .inputs
                        .get(*input)
                        .is_some_and(|input| input.constant.is_some()),
                    // A view is of a row before it, never of itself or a
                    // later one.
                    Owner::Row { row } => *row < index,
                },
            };
            ensure!(
                in_range,
                "row {:?} is stored in a slot, constant or row the code object does not have",
                row.name
            );
        }
        match &self.program {
            None => ensure!(
                self.rows.is_empty(),
                "a code object with no program states {} output rows",
                self.rows.len()
            ),
            Some(program) => {
                // Each row the caller provides memory for is written where
                // the tables say: the program's own output record names the
                // row's buffer.
                for (slot, row) in self.rows.iter().enumerate() {
                    let output = program
                        .output(slot)
                        .with_context(|| format!("row {:?}", row.name))?;
                    let written = program
                        .buffer(&output.buffer)
                        .with_context(|| format!("row {:?}", row.name))?
                        .lit;
                    ensure!(
                        written == Some(row.buffer),
                        "row {:?} is bound on buffer {}, and the program writes it on {written:?}",
                        row.name,
                        row.buffer
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use luminal::prelude::Symbol;
    use luminal_torch_code::{Check, Leaf, Literal, SlotAction, Source};
    use luminal_torch_slots::SlotKind;

    use super::*;

    fn sample() -> Code {
        let n = Symbol::new("s0");
        let shape = vec![IntExpr::from(n), IntExpr::from(4i64)];
        Code {
            format: FORMAT,
            boundary: Boundary {
                slots: vec![SlotKind::Tensor, SlotKind::Int, SlotKind::Tensor],
                slot_names: vec!["x".into(), "n".into(), "unused".into()],
                actions: vec![
                    SlotAction::Read { input: 0 },
                    SlotAction::Dimension { symbol: n },
                    SlotAction::Skip,
                ],
                leaves: vec![
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
            },
            inputs: vec![
                Input {
                    name: "x".into(),
                    buffer: 1,
                    shape: shape.clone(),
                    written: true,
                    constant: None,
                },
                Input {
                    name: "_tensor_constant0".into(),
                    buffer: 2,
                    shape: vec![IntExpr::from(2i64)],
                    written: false,
                    constant: Some(Constant {
                        dtype: TorchDtype::Int32,
                        shape: vec![2],
                        strides: vec![1],
                        bytes: vec![0, 1, 254, 255, 0, 0, 0, 0],
                    }),
                },
            ],
            rows: vec![],
            program: None,
        }
    }

    fn row(storage: Storage) -> Row {
        Row {
            name: "add".into(),
            dtype: TorchDtype::Float32,
            shape: vec![IntExpr::from(Symbol::new("s0")), IntExpr::from(4i64)],
            strides: Strides::Elements(vec![IntExpr::from(4i64), IntExpr::from(1i64)]),
            buffer: 3,
            storage,
        }
    }

    /// The byte form reads back as the same tables, and writing it again
    /// gives the same bytes: nothing in a code object depends on the process
    /// that wrote it.
    #[test]
    fn a_code_object_round_trips_through_its_bytes() {
        let code = sample();
        let bytes = code.to_bytes().unwrap();
        let read = Code::from_bytes(&bytes).unwrap();
        assert_eq!(read.to_bytes().unwrap(), bytes);
        assert_eq!(read.boundary.actions, code.boundary.actions);
        assert_eq!(read.boundary.dims, code.boundary.dims);
        let constant = read.inputs[1].constant.as_ref().unwrap();
        assert_eq!(constant.bytes, [0, 1, 254, 255, 0, 0, 0, 0]);
    }

    /// A constant is spelled in hex, and a dtype by the name torch gives it.
    #[test]
    fn the_saved_form_is_readable() {
        let text = String::from_utf8(sample().to_bytes().unwrap()).unwrap();
        assert!(text.contains(r#""bytes":"0001feff00000000""#), "{text}");
        assert!(text.contains(r#""dtype":"int32""#), "{text}");
        // The boundary tables sit at the top level, beside the backend's.
        let saved: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            saved["slots"],
            serde_json::json!(["tensor", "int", "tensor"])
        );
        assert_eq!(
            saved["actions"][0],
            serde_json::json!({"read": {"input": 0}})
        );
        assert!(saved.get("boundary").is_none(), "{text}");
    }

    #[test]
    fn another_format_is_refused_by_number() {
        let mut code = sample();
        code.format = FORMAT + 1;
        let error = Code::from_bytes(&code.to_bytes().unwrap()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("format {}", FORMAT + 1)),
            "{error}"
        );
    }

    /// Tables that index outside one another are refused at load, so a call
    /// never has to check.
    #[test]
    fn a_code_object_whose_tables_disagree_is_refused_at_load() {
        let refused = |code: Code| {
            Code::from_bytes(&serde_json::to_vec(&code).unwrap())
                .unwrap_err()
                .to_string()
        };

        let mut code = sample();
        code.boundary.leaves.push(Leaf::Row {
            row: 7,
            scalar: false,
        });
        assert!(refused(code).contains("output leaf 4"));

        let mut code = sample();
        code.boundary.actions.pop();
        assert!(refused(code).contains("3 slots and 2"));

        // A slot reads an argument, never a constant the program holds.
        let mut code = sample();
        code.boundary.actions[0] = SlotAction::Read { input: 1 };
        assert!(refused(code).contains("slot 0 reads input 1"));

        // A row needs a program to write it.
        let mut code = sample();
        code.rows.push(row(Storage::Fresh));
        assert!(refused(code).contains("no program states 1 output rows"));

        // A view is of a row before it.
        let mut code = sample();
        code.rows.push(row(Storage::View {
            of: Owner::Row { row: 0 },
            offset: 0,
        }));
        assert!(refused(code).contains("row \"add\" is stored in"));
    }
}
