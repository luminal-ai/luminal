//! The reference backend's code object: one compiled graph as plain data.
//!
//! A code object is everything a call needs and nothing the compiler needed.
//! It holds no parsed program, no translated graph and no search state. The
//! compiler builds one at the end of search and hands it over as bytes
//! ([`Code::to_bytes`]); every executable, in process or loaded from a file,
//! is built from those bytes ([`Code::from_bytes`]).

use anyhow::{Context, Result, bail};
use luminal::prelude::IntExpr;
use luminal_torch_code::{Boundary, hex};
use luminal_torch_slots::TorchDtype;
use serde::{Deserialize, Serialize};

/// The version of the byte form. A program saved under another version is
/// refused by number, never read as this one.
pub const FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Code {
    pub format: u32,
    /// Torch's side of a call, which every backend states the same way:
    /// the argument slots, the output leaves, the conditions torch asserted
    /// and the value each dimension starts at. A slot that is read is
    /// staged: its elements are copied to the host buffer of its input.
    #[serde(flatten)]
    pub boundary: Boundary,
    /// The tensors the program reads, each on its own buffer: the arguments
    /// it reads and the constants it holds.
    pub inputs: Vec<Input>,
    /// The tensors the program writes, in row order.
    pub rows: Vec<Row>,
    pub memory_budget_bytes: usize,
    /// The searched plan, in the reference runtime's own saved form. `None`
    /// for a graph that computes nothing: it forwards arguments, sizes and
    /// constants only.
    pub plan: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Input {
    /// The graph name, which every refusal leads with.
    pub name: String,
    pub dtype: TorchDtype,
    /// One extent per axis: a literal, or an expression over dimensions.
    pub shape: Vec<IntExpr>,
    /// The runtime buffer the tensor's elements are staged on.
    pub buffer: i64,
    /// For a constant the program holds: its elements, row-major. An
    /// argument has none; a call supplies it.
    #[serde(with = "hex::option", default, skip_serializing_if = "Option::is_none")]
    pub constant: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub name: String,
    pub dtype: TorchDtype,
    pub shape: Vec<IntExpr>,
    /// The runtime buffer the row is read from after a run.
    pub buffer: i64,
    /// For a mutation: the argument slot whose tensor the row is written
    /// back into. The caller's own tensor is then the row's value.
    pub writes_back: Option<usize>,
}

impl Code {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).context("writing a reference code object")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let code: Code = luminal_torch_code::bytes::read(bytes, FORMAT, "reference")?;
        code.validate()?;
        Ok(code)
    }

    /// What must hold for the tables to index one another. Checked once at
    /// load, so a call never indexes out of range.
    fn validate(&self) -> Result<()> {
        let arguments: Vec<bool> = self
            .inputs
            .iter()
            .map(|input| input.constant.is_none())
            .collect();
        self.boundary.validate(&arguments, self.rows.len())?;
        for row in &self.rows {
            if let Some(slot) = row.writes_back
                && slot >= self.boundary.slots.len()
            {
                bail!(
                    "row {:?} writes back into slot {slot}, which does not exist",
                    row.name
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use luminal::prelude::Symbol;
    use luminal_torch_code::{Leaf, SlotAction};
    use luminal_torch_slots::SlotKind;

    use super::*;

    fn sample() -> Code {
        let n = Symbol::new("s0");
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
                    Leaf::Row {
                        row: 0,
                        scalar: false,
                    },
                    Leaf::Argument { slot: 2 },
                ],
                checks: Vec::new(),
                dims: vec![(n, 3)],
            },
            inputs: vec![
                Input {
                    name: "x".into(),
                    dtype: TorchDtype::Float32,
                    shape: vec![IntExpr::from(n), IntExpr::from(4i64)],
                    buffer: 1,
                    constant: None,
                },
                Input {
                    name: "_tensor_constant0".into(),
                    dtype: TorchDtype::Int64,
                    shape: vec![IntExpr::from(1i64)],
                    buffer: 2,
                    constant: Some(vec![0, 1, 254, 255, 0, 0, 0, 0]),
                },
            ],
            rows: vec![Row {
                name: "add".into(),
                dtype: TorchDtype::Float32,
                shape: vec![IntExpr::from(n), IntExpr::from(4i64)],
                buffer: 3,
                writes_back: Some(0),
            }],
            memory_budget_bytes: 1 << 20,
            plan: None,
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
        assert_eq!(read.inputs[1].constant, code.inputs[1].constant);
    }

    /// The boundary tables sit at the top level of the saved form beside
    /// the backend's own, a constant is spelled in hex, and a dtype by the
    /// name torch gives it.
    #[test]
    fn the_saved_form_is_readable() {
        let text = String::from_utf8(sample().to_bytes().unwrap()).unwrap();
        assert!(text.contains(r#""constant":"0001feff00000000""#), "{text}");
        assert!(text.contains(r#""dtype":"float32""#), "{text}");
        let saved: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            saved["slots"],
            serde_json::json!(["tensor", "int", "tensor"])
        );
        assert!(saved.get("boundary").is_none(), "{text}");
        for dtype in TorchDtype::ALL {
            assert_eq!(
                serde_json::to_string(&dtype).unwrap(),
                format!("\"{}\"", dtype.name())
            );
        }
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
        let mut code = sample();
        code.boundary.leaves.push(Leaf::Row {
            row: 7,
            scalar: false,
        });
        let error = Code::from_bytes(&code.to_bytes().unwrap()).unwrap_err();
        assert!(error.to_string().contains("output leaf 2"), "{error}");

        let mut code = sample();
        code.rows[0].writes_back = Some(9);
        let error = Code::from_bytes(&code.to_bytes().unwrap()).unwrap_err();
        assert!(error.to_string().contains("slot 9"), "{error}");
    }
}
