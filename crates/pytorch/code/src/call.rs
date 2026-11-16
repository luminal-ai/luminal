//! The steps of a call that read only the boundary tables: the arguments,
//! slot by slot, and the output leaves.
//!
//! Between the two a backend does what is its own: it reads each tensor its
//! way, states the dimensions, runs its program and produces its rows.

use luminal::prelude::{DynMap, IntExpr, Symbol};
use luminal_torch_slots::{TensorInfo, read_int, read_tensor};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyList, PyString};

use crate::boundary::{Boundary, Leaf, Literal, SlotAction, Source};

/// A Rust error as the Python error a call raises. A Python error that
/// passed through Rust is raised again as itself.
pub fn to_py(error: anyhow::Error) -> PyErr {
    if let Some(python) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<PyErr>())
    {
        return Python::attach(|py| python.clone_ref(py));
    }
    PyRuntimeError::new_err(format!("{error:#}"))
}

/// The name of a value's type, for a refusal.
pub fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "object".to_string())
}

/// One tensor argument the program reads.
pub struct TensorArgument {
    /// The argument slot it was passed in.
    pub slot: usize,
    /// The input of the backend's input table it is read as.
    pub input: usize,
    pub info: TensorInfo,
}

/// What a call's arguments state, read slot by slot.
pub struct Arguments {
    /// The tensors the program reads, in slot order.
    pub tensors: Vec<TensorArgument>,
    /// The dimensions integer arguments define, in slot order.
    pub integers: Vec<(Symbol, usize)>,
}

impl Boundary {
    /// Read every slot of a call. Nothing is filtered: an argument the
    /// program does not read is not looked at. `runtime` names the package
    /// in the refusal of a call with the wrong number of arguments.
    pub fn read_arguments(&self, args: &Bound<'_, PyList>, runtime: &str) -> PyResult<Arguments> {
        if args.len() != self.slots.len() {
            return Err(PyRuntimeError::new_err(format!(
                "{runtime} expected {} inputs, got {}",
                self.slots.len(),
                args.len()
            )));
        }
        let mut tensors = Vec::new();
        let mut integers = Vec::new();
        for (slot, action) in self.actions.iter().enumerate() {
            let name = &self.slot_names[slot];
            match action {
                SlotAction::Read { input } => {
                    let value = args.get_item(slot)?;
                    let info = read_tensor(&value).map_err(|_| {
                        PyTypeError::new_err(format!(
                            "argument '{name}' is a tensor slot and was called with a {}",
                            type_name(&value)
                        ))
                    })?;
                    tensors.push(TensorArgument {
                        slot,
                        input: *input,
                        info,
                    });
                }
                SlotAction::Dimension { symbol } => {
                    let value = args.get_item(slot)?;
                    let integer = read_int(&value).map_err(|_| {
                        PyTypeError::new_err(format!(
                            "argument '{name}' is an integer slot and was called with a {}",
                            type_name(&value)
                        ))
                    })?;
                    let extent = usize::try_from(integer).map_err(|_| {
                        PyRuntimeError::new_err(format!(
                            "integer argument '{name}' is {integer}: it defines the symbol \
                             {symbol}, and a symbol's value is unsigned in this runtime \
                             (LUM-1337)"
                        ))
                    })?;
                    integers.push((*symbol, extent));
                }
                SlotAction::Skip => {}
            }
        }
        Ok(Arguments { tensors, integers })
    }

    /// The value of every output leaf of a call, in torch's order. `rows`
    /// holds the tensor of each row of the backend's row table.
    pub fn outputs<'py>(
        &self,
        args: &Bound<'py, PyList>,
        rows: &[Bound<'py, PyAny>],
    ) -> PyResult<Bound<'py, PyList>> {
        let py = args.py();
        let mut values: Vec<Bound<'py, PyAny>> = Vec::with_capacity(self.leaves.len());
        for leaf in &self.leaves {
            values.push(match leaf {
                Leaf::Row { row, scalar } => {
                    let tensor = rows[*row].clone();
                    // A Python scalar the graph returns was boxed into a
                    // rank-zero tensor for the program.
                    if *scalar {
                        tensor.call_method0("item")?
                    } else {
                        tensor
                    }
                }
                Leaf::Argument { slot } => args.get_item(*slot)?,
                Leaf::Integer { expr, sources } => self
                    .scalar(args, expr, sources)?
                    .into_pyobject(py)?
                    .into_any(),
                Leaf::Bool { expr, sources } => {
                    PyBool::new(py, self.scalar(args, expr, sources)? != 0)
                        .to_owned()
                        .into_any()
                }
                Leaf::Literal(literal) => match literal {
                    Literal::None => py.None().into_bound(py),
                    Literal::Bool(value) => PyBool::new(py, *value).to_owned().into_any(),
                    Literal::Int(value) => value.into_pyobject(py)?.into_any(),
                    Literal::Float(value) => PyFloat::new(py, *value).into_any(),
                    Literal::Str(value) => PyString::new(py, value).into_any(),
                },
            });
        }
        PyList::new(py, values)
    }

    /// A symbolic scalar leaf at this call: torch's expression with each
    /// dimension read from the argument that states it.
    fn scalar(
        &self,
        args: &Bound<'_, PyList>,
        expr: &IntExpr,
        sources: &[Source],
    ) -> PyResult<usize> {
        let mut stated = DynMap::default();
        for source in sources {
            let argument = args.get_item(source.slot)?;
            let value = match source.axis {
                None => read_int(&argument)?,
                Some(axis) => *read_tensor(&argument)?.shape.get(axis).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "argument '{}' has no axis {axis} to read dimension {} from",
                        self.slot_names[source.slot], source.symbol
                    ))
                })?,
            };
            let value = usize::try_from(value).map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "dimension {} was stated as {value}; a dimension is not negative",
                    source.symbol
                ))
            })?;
            stated.insert(source.symbol, value);
        }
        expr.exec(&stated)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unbound AOT output dimension: {expr}")))
    }
}
